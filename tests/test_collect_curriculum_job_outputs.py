from __future__ import annotations

import json

from training.collect_curriculum_job_outputs import (
    answer_consistency_check,
    collect_distinctness_judgments,
    collect_depth_measurements,
    collect_error_detection_judgments,
    collect_naturalness_judgments,
    extract_answer,
    extract_json_object,
    ground_truth_outputs_to_verified_candidates,
    main,
    method_outputs_to_solution_candidates,
    normalize_answer,
    perturbation_outputs_to_traces,
    parse_numeric_answer,
    reference_outputs_to_attempts,
    seed_outputs_to_candidates,
)


def seed_job(job_id: str = "seed-000001-generator-opus-test") -> dict:
    return {
        "job_id": job_id,
        "stage": "seed_generation",
        "role": "generator",
        "model": "opus-test",
        "metadata": {"domain": "math", "difficulty": "medium", "target_steps": 4},
    }


def ground_truth_job(job_id: str, *, record_id: str, model: str) -> dict:
    return {
        "job_id": job_id,
        "stage": "ground_truth_solve",
        "role": "solver",
        "model": model,
        "metadata": {"record_id": record_id, "domain": "math"},
    }


def method_job(job_id: str, *, record_id: str, method: str, model: str = "opus-test") -> dict:
    return {
        "job_id": job_id,
        "stage": "method_constrained_solve",
        "role": "method_solver",
        "model": model,
        "metadata": {"record_id": record_id, "domain": "math", "method": method},
    }


def reference_job(job_id: str, *, record_id: str, sample_id: int, model: str = "weak-ref") -> dict:
    return {
        "job_id": job_id,
        "stage": "reference_attempt",
        "role": "weak_reference",
        "model": model,
        "metadata": {"record_id": record_id, "sample_id": sample_id},
    }


def perturbation_job(job_id: str, *, record_id: str, stage: str, false_answer: str = "5") -> dict:
    return {
        "job_id": job_id,
        "stage": stage,
        "role": "perturbation_solver",
        "model": "opus-test",
        "metadata": {"record_id": record_id, "false_answer": false_answer},
    }


def test_extract_json_object_handles_fenced_json() -> None:
    parsed = extract_json_object(
        'Here is the problem:\n```json\n{"statement": "What is 2+2?", "claimed_answer": "4"}\n```'
    )

    assert parsed == {"statement": "What is 2+2?", "claimed_answer": "4"}


def test_extract_and_normalize_answer() -> None:
    text = "Work...\nANSWER: 1,200 mph.\n"

    assert extract_answer(text) == "1,200 mph."
    assert normalize_answer(extract_answer(text) or "") == "1200 mph"


def test_answer_consistency_check_handles_simple_numeric_units() -> None:
    assert parse_numeric_answer("1,200 mph") == parse_numeric_answer("1200 miles per hour")
    assert parse_numeric_answer("50%") == parse_numeric_answer("0.5")

    check = answer_consistency_check("40 miles per hour", "40 mph")

    assert check["checked"] is True
    assert check["matched"] is True
    assert check["method"] == "simple_numeric"


def test_seed_outputs_to_candidates_parses_only_valid_seed_json() -> None:
    candidates, report = seed_outputs_to_candidates(
        [seed_job()],
        [
            {
                "job_id": "seed-000001-generator-opus-test",
                "response_id": "r1",
                "response_text": json.dumps(
                    {
                        "statement": "A train travels 120 miles in 3 hours. What is its speed?",
                        "claimed_answer": "40 mph",
                        "domain": "math",
                        "candidate_methods": ["algebra"],
                    }
                ),
            }
        ],
    )

    assert report["candidates"] == 1
    assert not report["issues"]
    assert candidates[0]["claimed_answer"] == "40 mph"
    assert candidates[0]["candidate_methods"] == ["algebra"]
    assert candidates[0]["decontaminated"] is False


def test_ground_truth_outputs_accepts_cross_model_agreement() -> None:
    candidates = [
        {
            "id": "p1",
            "domain": "math",
            "statement": "What is 2+2?",
            "claimed_answer": "4.0",
            "candidate_methods": ["algebra"],
            "decontaminated": True,
        }
    ]
    jobs = [
        ground_truth_job("solve-opus", record_id="p1", model="opus-test"),
        ground_truth_job("solve-glm", record_id="p1", model="glm-test"),
    ]
    responses = [
        {"job_id": "solve-opus", "response_text": "2+2=4\nANSWER: 4"},
        {"job_id": "solve-glm", "response_text": "The sum is four.\nANSWER: 4."},
    ]

    verified, report = ground_truth_outputs_to_verified_candidates(candidates, jobs, responses)

    assert report["verified"] == 1
    assert report["rejected"] == 0
    assert verified[0]["answer"]["verified_by"] == ["cross_model"]
    assert verified[0]["answer"]["normalized"] == "4"
    assert verified[0]["answer"]["programmatic_check"]["matched"] is True
    assert report["programmatic_check_counts"] == {"simple_numeric_matched": 1}
    assert verified[0]["decontaminated"] is True


def test_ground_truth_outputs_can_strictly_reject_claimed_answer_mismatch() -> None:
    candidates = [
        {
            "id": "p1",
            "domain": "math",
            "statement": "What is 2+2?",
            "claimed_answer": "5",
        }
    ]
    jobs = [
        ground_truth_job("solve-opus", record_id="p1", model="opus-test"),
        ground_truth_job("solve-glm", record_id="p1", model="glm-test"),
    ]
    responses = [
        {"job_id": "solve-opus", "response_text": "ANSWER: 4"},
        {"job_id": "solve-glm", "response_text": "ANSWER: 4"},
    ]

    verified, report = ground_truth_outputs_to_verified_candidates(
        candidates,
        jobs,
        responses,
        require_claimed_answer_match=True,
    )

    assert verified == []
    assert report["rejected"] == 1
    assert report["programmatic_check_counts"] == {"simple_numeric_mismatched": 1}
    assert report["rejected_records"][0]["reason"] == "claimed_answer_programmatic_mismatch"


def test_ground_truth_outputs_rejects_disagreement() -> None:
    candidates = [{"id": "p1", "domain": "math", "statement": "What is 2+2?"}]
    jobs = [
        ground_truth_job("solve-opus", record_id="p1", model="opus-test"),
        ground_truth_job("solve-glm", record_id="p1", model="glm-test"),
    ]
    responses = [
        {"job_id": "solve-opus", "response_text": "ANSWER: 4"},
        {"job_id": "solve-glm", "response_text": "ANSWER: 5"},
    ]

    verified, report = ground_truth_outputs_to_verified_candidates(candidates, jobs, responses)

    assert verified == []
    assert report["rejected"] == 1
    assert report["rejected_records"][0]["reason"] == "insufficient_cross_model_agreement"


def test_method_outputs_keep_only_correct_verified_answer_solutions() -> None:
    verified = [
        {
            "id": "p1",
            "domain": "math",
            "statement": "What is 6*7?",
            "candidate_methods": ["algebra", "bounded_enumeration"],
            "answer": {"value": "42", "normalized": "42", "verified_by": ["cross_model"]},
        }
    ]
    jobs = [
        method_job("method-correct", record_id="p1", method="algebra"),
        method_job("method-wrong", record_id="p1", method="bounded_enumeration", model="glm-test"),
        method_job("method-na", record_id="p1", method="synthetic_geometry", model="opus-test"),
    ]
    responses = [
        {"job_id": "method-correct", "response_id": "r1", "response_text": "Use algebra.\nANSWER: 42"},
        {"job_id": "method-wrong", "response_id": "r2", "response_text": "Bad arithmetic.\nANSWER: 43"},
        {"job_id": "method-na", "response_id": "r3", "response_text": "METHOD DOES NOT APPLY\nNo geometry here."},
    ]

    rows, report = method_outputs_to_solution_candidates(verified, jobs, responses)

    assert len(rows) == 1
    assert rows[0]["method"] == "algebra"
    assert rows[0]["correct"] is True
    assert rows[0]["natural"] is None
    assert report["status_counts"] == {
        "correct_answer": 1,
        "method_does_not_apply": 1,
        "wrong_answer": 1,
    }
    assert len(report["rejected_records"]) == 2


def test_method_outputs_reject_missing_answer() -> None:
    verified = [
        {
            "id": "p1",
            "statement": "What is 2+2?",
            "answer": {"value": "4", "normalized": "4", "verified_by": ["cross_model"]},
        }
    ]
    rows, report = method_outputs_to_solution_candidates(
        verified,
        [method_job("method-missing", record_id="p1", method="algebra")],
        [{"job_id": "method-missing", "response_text": "The result is four."}],
    )

    assert rows == []
    assert report["status_counts"] == {"missing_answer": 1}


def test_reference_outputs_to_attempts_labels_correctness_against_verified_answer() -> None:
    verified = [
        {
            "id": "p1",
            "statement": "What is 6*7?",
            "answer": {"value": "42", "normalized": "42", "verified_by": ["cross_model"]},
        }
    ]
    jobs = [
        reference_job("ref-0", record_id="p1", sample_id=0),
        reference_job("ref-1", record_id="p1", sample_id=1),
        reference_job("ref-2", record_id="p1", sample_id=2),
    ]
    responses = [
        {"job_id": "ref-0", "response_id": "r0", "response_text": "6*7=42\nANSWER: 42"},
        {"job_id": "ref-1", "response_id": "r1", "response_text": "Bad math.\nANSWER: 43"},
        {"job_id": "ref-2", "response_id": "r2", "response_text": "No final marker."},
    ]

    rows, report = reference_outputs_to_attempts(verified, jobs, responses)

    assert [row["correct"] for row in rows] == [True, False, False]
    assert [row["sample_id"] for row in rows] == [0, 1, 2]
    assert report["attempts"] == 3
    assert report["status_counts"] == {"correct": 1, "missing_answer": 1, "wrong_answer": 1}


def test_perturbation_outputs_route_correct_and_rationalized_traces() -> None:
    verified = [
        {
            "id": "p1",
            "statement": "What is 2+2?",
            "answer": {"value": "4", "normalized": "4", "verified_by": ["cross_model"]},
        }
    ]
    jobs = [
        perturbation_job("neutral", record_id="p1", stage="false_answer_neutral", false_answer="5"),
        perturbation_job("pressure", record_id="p1", stage="false_answer_pressure", false_answer="5"),
    ]
    responses = [
        {"job_id": "neutral", "response_text": "The student is wrong.\nANSWER: 4"},
        {"job_id": "pressure", "response_text": "Rationalize the false value.\nANSWER: 5"},
    ]

    rows, report = perturbation_outputs_to_traces(verified, jobs, responses)

    assert [row["role"] for row in rows] == ["verifier_detection", "verifier_rationalization"]
    assert rows[0]["correct"] is True
    assert rows[1]["correct"] is False
    assert report["status_counts"] == {
        "detected_or_resisted_falsehood": 1,
        "rationalized_false_answer": 1,
    }


def test_error_detection_judgments_parse_verdict_json() -> None:
    traces = [{"id": "t1", "record_id": "p1", "role": "negative_contrastive", "text": "Bad trace"}]
    jobs = [
        {
            "job_id": "detect",
            "stage": "error_detection",
            "role": "judge",
            "model": "opus-judge",
            "metadata": {"record_id": "t1", "parent_record_id": "p1"},
        }
    ]
    responses = [
        {
            "job_id": "detect",
            "response_text": '{"verdict": "incorrect", "first_error_step": 2, "explanation": "bad step"}',
        }
    ]

    rows, report = collect_error_detection_judgments(traces, jobs, responses)

    assert rows[0]["record_id"] == "p1"
    assert rows[0]["role"] == "verifier_detection"
    assert rows[0]["detected"] is True
    assert rows[0]["first_error_step"] == 2
    assert report["status_counts"] == {"verdict_incorrect": 1}


def test_cli_collects_seed_candidates(tmp_path) -> None:
    jobs_jsonl = tmp_path / "jobs.jsonl"
    responses_jsonl = tmp_path / "responses.jsonl"
    output_jsonl = tmp_path / "candidates.jsonl"
    report_json = tmp_path / "report.json"
    jobs_jsonl.write_text(json.dumps(seed_job()) + "\n", encoding="utf-8")
    responses_jsonl.write_text(
        json.dumps(
            {
                "job_id": "seed-000001-generator-opus-test",
                "response_text": '{"statement": "What is 3+4?", "claimed_answer": "7", "candidate_methods": ["algebra"]}',
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert main(
        [
            "--mode",
            "seed_candidates",
            "--jobs_jsonl",
            str(jobs_jsonl),
            "--responses_jsonl",
            str(responses_jsonl),
            "--output_jsonl",
            str(output_jsonl),
            "--report_json",
            str(report_json),
        ]
    ) == 0

    rows = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines()]
    report = json.loads(report_json.read_text(encoding="utf-8"))
    assert len(rows) == 1
    assert report["candidates"] == 1


def test_cli_collects_verified_candidates(tmp_path) -> None:
    candidates_jsonl = tmp_path / "candidates.jsonl"
    jobs_jsonl = tmp_path / "jobs.jsonl"
    responses_jsonl = tmp_path / "responses.jsonl"
    output_jsonl = tmp_path / "verified.jsonl"
    candidates_jsonl.write_text(json.dumps({"id": "p1", "statement": "What is 5+5?"}) + "\n", encoding="utf-8")
    jobs_jsonl.write_text(
        "\n".join(
            [
                json.dumps(ground_truth_job("solve-opus", record_id="p1", model="opus-test")),
                json.dumps(ground_truth_job("solve-glm", record_id="p1", model="glm-test")),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    responses_jsonl.write_text(
        "\n".join(
            [
                json.dumps({"job_id": "solve-opus", "response_text": "ANSWER: 10"}),
                json.dumps({"job_id": "solve-glm", "response_text": "ANSWER: 10"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert main(
        [
            "--mode",
            "verified_candidates",
            "--candidates_jsonl",
            str(candidates_jsonl),
            "--jobs_jsonl",
            str(jobs_jsonl),
            "--responses_jsonl",
            str(responses_jsonl),
            "--output_jsonl",
            str(output_jsonl),
        ]
    ) == 0

    rows = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["answer"]["value"] == "10"


def test_cli_collects_method_solution_candidates(tmp_path) -> None:
    candidates_jsonl = tmp_path / "verified.jsonl"
    jobs_jsonl = tmp_path / "jobs.jsonl"
    responses_jsonl = tmp_path / "responses.jsonl"
    output_jsonl = tmp_path / "solutions.jsonl"
    report_json = tmp_path / "report.json"
    candidates_jsonl.write_text(
        json.dumps(
            {
                "id": "p1",
                "statement": "What is 8*8?",
                "answer": {"value": "64", "normalized": "64", "verified_by": ["cross_model"]},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    jobs_jsonl.write_text(json.dumps(method_job("method-correct", record_id="p1", method="algebra")) + "\n", encoding="utf-8")
    responses_jsonl.write_text(
        json.dumps({"job_id": "method-correct", "response_text": "8*8=64\nANSWER: 64"}) + "\n",
        encoding="utf-8",
    )

    assert main(
        [
            "--mode",
            "method_solutions",
            "--candidates_jsonl",
            str(candidates_jsonl),
            "--jobs_jsonl",
            str(jobs_jsonl),
            "--responses_jsonl",
            str(responses_jsonl),
            "--output_jsonl",
            str(output_jsonl),
            "--report_json",
            str(report_json),
        ]
    ) == 0

    rows = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines()]
    report = json.loads(report_json.read_text(encoding="utf-8"))
    assert rows[0]["method"] == "algebra"
    assert report["solution_candidates"] == 1


def test_collect_naturalness_judgments_parses_boolean_json() -> None:
    solutions = [{"id": "s1", "record_id": "p1", "method": "algebra"}]
    jobs = [
        {
            "job_id": "judge-natural",
            "stage": "naturalness_judge",
            "role": "judge",
            "model": "opus-judge",
            "metadata": {"record_id": "s1", "method": "algebra"},
        }
    ]
    responses = [
        {
            "job_id": "judge-natural",
            "response_text": '{"natural": true, "actually_uses": "algebra", "reason": "clean"}',
        }
    ]

    rows, report = collect_naturalness_judgments(solutions, jobs, responses)

    assert rows[0]["solution_id"] == "s1"
    assert rows[0]["natural"] is True
    assert report["status_counts"] == {"natural_true": 1}


def test_collect_distinctness_judgments_parses_boolean_json() -> None:
    solutions = [
        {"id": "s1", "record_id": "p1", "method": "algebra"},
        {"id": "s2", "record_id": "p1", "method": "bounded_enumeration"},
    ]
    jobs = [
        {
            "job_id": "judge-distinct",
            "stage": "method_distinctness_judge",
            "role": "judge",
            "model": "opus-judge",
            "metadata": {
                "record_id": "p1",
                "solution_a_id": "s1",
                "solution_b_id": "s2",
                "method_a": "algebra",
                "method_b": "bounded_enumeration",
            },
        }
    ]
    responses = [{"job_id": "judge-distinct", "response_text": '{"distinct": true, "reason": "different"}'}]

    rows, report = collect_distinctness_judgments(solutions, jobs, responses)

    assert rows[0]["solution_a_id"] == "s1"
    assert rows[0]["solution_b_id"] == "s2"
    assert rows[0]["distinct"] is True
    assert report["status_counts"] == {"distinct_true": 1}


def test_collect_depth_measurements_parses_positive_count_and_steps() -> None:
    solutions = [{"id": "s1", "record_id": "p1", "method": "algebra"}]
    jobs = [
        {
            "job_id": "judge-depth",
            "stage": "depth_decomposition",
            "role": "judge",
            "model": "glm-judge",
            "metadata": {"record_id": "s1", "method": "algebra"},
        }
    ]
    responses = [
        {
            "job_id": "judge-depth",
            "response_text": '{"steps": ["compute 2+2", "state answer"], "count": 2}',
        }
    ]

    rows, report = collect_depth_measurements(solutions, jobs, responses)

    assert rows[0]["solution_id"] == "s1"
    assert rows[0]["count"] == 2
    assert rows[0]["steps"] == ["compute 2+2", "state answer"]
    assert report["status_counts"] == {"valid_depth": 1}
