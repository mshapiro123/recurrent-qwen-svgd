from __future__ import annotations

import json

from training.collect_curriculum_job_outputs import (
    extract_answer,
    extract_json_object,
    ground_truth_outputs_to_verified_candidates,
    main,
    normalize_answer,
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


def test_extract_json_object_handles_fenced_json() -> None:
    parsed = extract_json_object(
        'Here is the problem:\n```json\n{"statement": "What is 2+2?", "claimed_answer": "4"}\n```'
    )

    assert parsed == {"statement": "What is 2+2?", "claimed_answer": "4"}


def test_extract_and_normalize_answer() -> None:
    text = "Work...\nANSWER: 1,200 mph.\n"

    assert extract_answer(text) == "1,200 mph."
    assert normalize_answer(extract_answer(text) or "") == "1200 mph"


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
    assert verified[0]["decontaminated"] is True


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

