"""Run a no-API, no-GPU fixture through the full curriculum data pipeline.

This is a regression smoke for the strong-model curriculum machinery. It uses
synthetic provider responses to exercise the same stages a real API runner will
use: jobs, raw responses, candidate collection, cross-model answer agreement,
measured difficulty, method-solution collection,
naturalness/distinctness/depth judgments, typed record assembly, and positive
SFT export.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.annotate_curriculum_difficulty import annotate_difficulty
from training.annotate_curriculum_false_answers import annotate_false_answers
from training.assemble_curriculum_records import assemble_curriculum_records
from training.build_curriculum_generation_jobs import (
    build_depth_jobs,
    build_distinctness_jobs,
    build_error_detection_jobs,
    build_ground_truth_jobs,
    build_method_solve_jobs,
    build_naturalness_jobs,
    build_perturbation_jobs,
    build_seed_jobs,
)
from training.collect_curriculum_job_outputs import (
    collect_distinctness_judgments,
    collect_depth_measurements,
    collect_error_detection_judgments,
    collect_naturalness_judgments,
    ground_truth_outputs_to_verified_candidates,
    method_outputs_to_solution_candidates,
    perturbation_outputs_to_traces,
    seed_outputs_to_candidates,
)
from training.prepare_curriculum_jsonl import convert_curriculum_records


GENERATOR_MODEL = "opus-fixture"
SOLVER_MODELS = ["opus-fixture", "glm-fixture"]
JUDGE_MODELS = ["opus-fixture", "glm-fixture"]
STATEMENT = "Find the area of a rectangle with side lengths 6 and 7."
ANSWER = "42"
METHODS = ["algebra", "bounded_enumeration"]


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def seed_responses(seed_jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "job_id": job["job_id"],
            "response_id": f"response-{job['job_id']}",
            "response_text": json.dumps(
                {
                    "statement": STATEMENT,
                    "claimed_answer": ANSWER,
                    "domain": "math",
                    "candidate_methods": METHODS,
                }
            ),
        }
        for job in seed_jobs
    ]


def ground_truth_responses(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "job_id": job["job_id"],
            "response_id": f"response-{job['job_id']}",
            "response_text": "The area of a rectangle is length times width: 6 * 7 = 42.\nANSWER: 42",
        }
        for job in jobs
    ]


def method_solution_text(method: str) -> str:
    if method == "algebra":
        return "Use the area formula A = length * width. So A = 6 * 7 = 42.\nANSWER: 42"
    return "Count 7 unit squares in each of 6 rows, giving 6 groups of 7: 42.\nANSWER: 42"


def method_responses(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for job in jobs:
        metadata = job.get("metadata") if isinstance(job.get("metadata"), dict) else {}
        method = str(metadata.get("method") or "")
        rows.append(
            {
                "job_id": job["job_id"],
                "response_id": f"response-{job['job_id']}",
                "response_text": method_solution_text(method),
            }
        )
    return rows


def naturalness_responses(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for job in jobs:
        metadata = job.get("metadata") if isinstance(job.get("metadata"), dict) else {}
        method = str(metadata.get("method") or "")
        rows.append(
            {
                "job_id": job["job_id"],
                "response_id": f"response-{job['job_id']}",
                "response_text": json.dumps(
                    {
                        "natural": True,
                        "actually_uses": method,
                        "reason": "The solution uses the requested method directly.",
                    }
                ),
            }
        )
    return rows


def distinctness_responses(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "job_id": job["job_id"],
            "response_id": f"response-{job['job_id']}",
            "response_text": json.dumps(
                {
                    "distinct": True,
                    "reason": "One solution uses a formula and the other counts rows of unit squares.",
                }
            ),
        }
        for job in jobs
    ]


def depth_for_method(method: str) -> tuple[list[str], int]:
    if method == "algebra":
        return ["Recall rectangle area formula.", "Multiply 6 by 7.", "State 42."], 3
    return ["View the rectangle as rows.", "Count 7 cells per row.", "Count 6 rows.", "Multiply 6 by 7.", "State 42."], 5


def depth_responses(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for job in jobs:
        metadata = job.get("metadata") if isinstance(job.get("metadata"), dict) else {}
        method = str(metadata.get("method") or "")
        steps, count = depth_for_method(method)
        rows.append(
            {
                "job_id": job["job_id"],
                "response_id": f"response-{job['job_id']}",
                "response_text": json.dumps({"steps": steps, "count": count}),
            }
        )
    return rows


def reference_attempts(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        record_id = str(candidate.get("id") or "")
        rows.extend(
            [
                {"record_id": record_id, "sample_id": 0, "correct": True},
                {"record_id": record_id, "sample_id": 1, "correct": False},
                {"record_id": record_id, "sample_id": 2, "correct": True},
                {"record_id": record_id, "sample_id": 3, "correct": True},
            ]
        )
    return rows


def perturbation_responses(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for job in jobs:
        metadata = job.get("metadata") if isinstance(job.get("metadata"), dict) else {}
        false_answer = str(metadata.get("false_answer") or "")
        if job["stage"] == "false_answer_neutral" and job.get("model") == SOLVER_MODELS[0]:
            text = "The proposed answer is not correct. The area is 6 * 7 = 42.\nANSWER: 42"
        elif job["stage"] == "false_answer_neutral":
            text = "A careless calculation gives 6 * 7 = 41.\nANSWER: 41"
        else:
            text = f"If we make a plausible arithmetic slip, the answer is {false_answer}.\nANSWER: {false_answer}"
        rows.append(
            {
                "job_id": job["job_id"],
                "response_id": f"response-{job['job_id']}",
                "response_text": text,
            }
        )
    return rows


def error_detection_responses(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "job_id": job["job_id"],
            "response_id": f"response-{job['job_id']}",
            "response_text": json.dumps(
                {
                    "verdict": "incorrect",
                    "first_error_step": 1,
                    "explanation": "The trace adds side lengths instead of multiplying for area.",
                }
            ),
        }
        for job in jobs
    ]


def run_fixture_pipeline(output_dir: str | Path, *, overwrite: bool = False) -> dict[str, Any]:
    out = Path(output_dir)
    if out.exists() and overwrite:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    seed_jobs = build_seed_jobs(
        models=[GENERATOR_MODEL],
        domains=["math"],
        difficulties=["medium"],
        target_steps=[4],
        count_per_combo=1,
    )
    seed_raw = seed_responses(seed_jobs)
    candidates, candidates_report = seed_outputs_to_candidates(seed_jobs, seed_raw, mark_decontaminated=True)

    ground_jobs = build_ground_truth_jobs(candidates, models=SOLVER_MODELS)
    ground_raw = ground_truth_responses(ground_jobs)
    verified, verified_report = ground_truth_outputs_to_verified_candidates(
        candidates,
        ground_jobs,
        ground_raw,
        mark_decontaminated=True,
    )

    difficulty_attempts = reference_attempts(verified)
    verified_with_difficulty, difficulty_rejected, difficulty_report = annotate_difficulty(
        verified,
        difficulty_attempts,
        reference_model="weak-fixture",
        min_samples=4,
    )
    verified_with_false_answers, false_answer_rejected, false_answer_report = annotate_false_answers(
        verified_with_difficulty
    )

    perturbation_jobs = build_perturbation_jobs(verified_with_false_answers, models=SOLVER_MODELS)
    perturbation_raw = perturbation_responses(perturbation_jobs)
    perturbation_traces, perturbation_report = perturbation_outputs_to_traces(
        verified_with_false_answers,
        perturbation_jobs,
        perturbation_raw,
    )

    error_detection_jobs = build_error_detection_jobs(perturbation_traces, models=JUDGE_MODELS)
    error_detection_raw = error_detection_responses(error_detection_jobs)
    error_detection, error_detection_report = collect_error_detection_judgments(
        perturbation_traces,
        error_detection_jobs,
        error_detection_raw,
    )

    method_jobs = build_method_solve_jobs(verified_with_false_answers, models=SOLVER_MODELS, fallback_methods=METHODS)
    method_raw = method_responses(method_jobs)
    method_solutions, method_report = method_outputs_to_solution_candidates(verified_with_false_answers, method_jobs, method_raw)

    natural_jobs = build_naturalness_jobs(method_solutions, models=JUDGE_MODELS)
    natural_raw = naturalness_responses(natural_jobs)
    naturalness, naturalness_report = collect_naturalness_judgments(method_solutions, natural_jobs, natural_raw)

    distinct_jobs = build_distinctness_jobs(method_solutions, models=JUDGE_MODELS)
    distinct_raw = distinctness_responses(distinct_jobs)
    distinctness, distinctness_report = collect_distinctness_judgments(method_solutions, distinct_jobs, distinct_raw)

    depth_jobs = build_depth_jobs(method_solutions, models=JUDGE_MODELS)
    depth_raw = depth_responses(depth_jobs)
    depth, depth_report = collect_depth_measurements(method_solutions, depth_jobs, depth_raw)

    records, records_report = assemble_curriculum_records(
        verified_with_false_answers,
        method_solutions,
        naturalness,
        depth,
        distinctness,
        [*perturbation_traces, *error_detection],
        min_natural_agree=2,
        min_distinct_agree=2,
        deep_threshold=5,
    )
    sft_rows, sft_report = convert_curriculum_records(records)

    artifacts: dict[str, Any] = {
        "seed_jobs": seed_jobs,
        "seed_responses": seed_raw,
        "candidates": candidates,
        "ground_truth_jobs": ground_jobs,
        "ground_truth_responses": ground_raw,
        "verified_candidates": verified,
        "difficulty_attempts": difficulty_attempts,
        "difficulty_rejected": difficulty_rejected,
        "verified_candidates_difficulty": verified_with_difficulty,
        "false_answer_rejected": false_answer_rejected,
        "verified_candidates_false_answers": verified_with_false_answers,
        "perturbation_jobs": perturbation_jobs,
        "perturbation_responses": perturbation_raw,
        "perturbation_traces": perturbation_traces,
        "error_detection_jobs": error_detection_jobs,
        "error_detection_responses": error_detection_raw,
        "error_detection_judgments": error_detection,
        "method_jobs": method_jobs,
        "method_responses": method_raw,
        "method_solution_candidates": method_solutions,
        "naturalness_jobs": natural_jobs,
        "naturalness_responses": natural_raw,
        "naturalness_judgments": naturalness,
        "distinctness_jobs": distinct_jobs,
        "distinctness_responses": distinct_raw,
        "distinctness_judgments": distinctness,
        "depth_jobs": depth_jobs,
        "depth_responses": depth_raw,
        "depth_measurements": depth,
        "typed_records": records,
        "positive_sft": sft_rows,
    }
    reports: dict[str, Any] = {
        "candidates": candidates_report,
        "verified_candidates": verified_report,
        "difficulty": difficulty_report,
        "false_answers": false_answer_report,
        "perturbation": perturbation_report,
        "error_detection": error_detection_report,
        "method_solutions": method_report,
        "naturalness": naturalness_report,
        "distinctness": distinctness_report,
        "depth": depth_report,
        "typed_records": records_report,
        "positive_sft": sft_report,
    }

    for name, rows in artifacts.items():
        write_jsonl(out / f"{name}.jsonl", rows)
    for name, report in reports.items():
        write_json(out / f"{name}_report.json", report)

    summary = {
        "output_dir": str(out),
        "typed_records": len(records),
        "positive_sft_rows": len(sft_rows),
        "mode_counts": records_report["mode_counts"],
        "reports": reports,
    }
    write_json(out / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="outputs/curriculum_fixture")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    summary = run_fixture_pipeline(args.output_dir, overwrite=args.overwrite)
    print(f"output_dir={summary['output_dir']}")
    print(f"typed_records={summary['typed_records']}")
    print(f"positive_sft_rows={summary['positive_sft_rows']}")
    print(f"mode_counts={summary['mode_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
