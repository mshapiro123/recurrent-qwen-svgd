"""Run the curriculum data pipeline from job/response artifacts.

This script is intentionally provider-neutral. It builds the next JSONL job
file, consumes response JSONL files when they exist, writes validated
intermediate artifacts, and stops cleanly at the next missing external artifact.
Use it in Colab or locally to avoid hand-assembling the wide/deep curriculum
pipeline in notebooks.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable


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
    build_reference_attempt_jobs,
    build_seed_jobs,
    validate_external_models,
)
from training.collect_curriculum_job_outputs import (
    collect_depth_measurements,
    collect_distinctness_judgments,
    collect_error_detection_judgments,
    collect_naturalness_judgments,
    ground_truth_outputs_to_verified_candidates,
    method_outputs_to_solution_candidates,
    perturbation_outputs_to_traces,
    reference_outputs_to_attempts,
    seed_outputs_to_candidates,
)
from training.decontaminate_curriculum_candidates import (
    annotate_candidates,
    build_reference_index,
)
from training.prepare_curriculum_jsonl import convert_curriculum_records


def split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_csv(value: str) -> list[int]:
    parsed = [int(item) for item in split_csv(value)]
    if not parsed:
        raise argparse.ArgumentTypeError("Expected at least one integer")
    return parsed


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"Line {line_no} in {path} is not a JSON object.")
        rows.append(row)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def existing_reference_paths(values: list[str], root: Path) -> list[Path]:
    if values:
        paths = [Path(value) for item in values for value in item.split(",") if value.strip()]
    else:
        paths = [root / "eval" / "smoke_exact_tasks_v2.jsonl", root / "eval" / "smoke_mcq_tasks.jsonl"]
    resolved = [path if path.is_absolute() else root / path for path in paths]
    return [path for path in resolved if path.exists()]


def artifact_paths(work_dir: Path) -> dict[str, Path]:
    return {
        "jobs_seed": work_dir / "jobs_seed.jsonl",
        "responses_seed": work_dir / "responses_seed.jsonl",
        "candidates": work_dir / "candidates.jsonl",
        "candidates_report": work_dir / "candidates_report.json",
        "candidates_decontaminated": work_dir / "candidates_decontaminated.jsonl",
        "candidates_contaminated": work_dir / "candidates_contaminated.jsonl",
        "decontam_report": work_dir / "decontam_report.json",
        "jobs_ground_truth": work_dir / "jobs_ground_truth.jsonl",
        "responses_ground_truth": work_dir / "responses_ground_truth.jsonl",
        "verified_candidates": work_dir / "verified_candidates.jsonl",
        "verified_candidates_report": work_dir / "verified_candidates_report.json",
        "jobs_reference_attempts": work_dir / "jobs_reference_attempts.jsonl",
        "responses_reference_attempts": work_dir / "responses_reference_attempts.jsonl",
        "reference_attempts": work_dir / "reference_attempts.jsonl",
        "reference_attempts_report": work_dir / "reference_attempts_report.json",
        "verified_candidates_difficulty": work_dir / "verified_candidates_difficulty.jsonl",
        "difficulty_report": work_dir / "difficulty_report.json",
        "verified_candidates_no_false_answer": work_dir / "verified_candidates_no_false_answer.jsonl",
        "verified_candidates_false_answers": work_dir / "verified_candidates_false_answers.jsonl",
        "false_answers_report": work_dir / "false_answers_report.json",
        "jobs_methods": work_dir / "jobs_methods.jsonl",
        "responses_methods": work_dir / "responses_methods.jsonl",
        "method_solution_candidates": work_dir / "method_solution_candidates.jsonl",
        "method_solutions_report": work_dir / "method_solutions_report.json",
        "jobs_perturbation": work_dir / "jobs_perturbation.jsonl",
        "responses_perturbation": work_dir / "responses_perturbation.jsonl",
        "perturbation_traces": work_dir / "perturbation_traces.jsonl",
        "perturbation_report": work_dir / "perturbation_report.json",
        "jobs_naturalness": work_dir / "jobs_naturalness.jsonl",
        "responses_naturalness": work_dir / "responses_naturalness.jsonl",
        "naturalness_judgments": work_dir / "naturalness_judgments.jsonl",
        "naturalness_report": work_dir / "naturalness_report.json",
        "jobs_distinctness": work_dir / "jobs_distinctness.jsonl",
        "responses_distinctness": work_dir / "responses_distinctness.jsonl",
        "distinctness_judgments": work_dir / "distinctness_judgments.jsonl",
        "distinctness_report": work_dir / "distinctness_report.json",
        "jobs_depth": work_dir / "jobs_depth.jsonl",
        "responses_depth": work_dir / "responses_depth.jsonl",
        "depth_measurements": work_dir / "depth_measurements.jsonl",
        "depth_report": work_dir / "depth_report.json",
        "jobs_error_detection": work_dir / "jobs_error_detection.jsonl",
        "responses_error_detection": work_dir / "responses_error_detection.jsonl",
        "error_detection_judgments": work_dir / "error_detection_judgments.jsonl",
        "error_detection_report": work_dir / "error_detection_report.json",
        "typed_records": work_dir / "typed_records.jsonl",
        "typed_records_report": work_dir / "typed_records_report.json",
        "positive_sft": work_dir / "positive_sft.jsonl",
        "positive_sft_report": work_dir / "positive_sft_report.json",
        "summary": work_dir / "summary.json",
    }


def stop_summary(
    *,
    work_dir: Path,
    status: str,
    next_action: str,
    artifacts: dict[str, Path],
    counts: dict[str, Any] | None = None,
    pending_responses: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    summary = {
        "kind": "curriculum_pipeline_from_artifacts",
        "work_dir": str(work_dir),
        "status": status,
        "next_action": next_action,
        "artifacts": {
            name: {"path": str(path), "exists": path.exists(), "lines": line_count(path) if path.exists() else 0}
            for name, path in artifacts.items()
        },
        "counts": counts or {},
    }
    if pending_responses:
        summary["pending_responses"] = pending_responses
    write_json(artifacts["summary"], summary)
    return summary


def line_count(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def response_missing(path: Path) -> bool:
    return not path.exists() or line_count(path) == 0


def response_incomplete(path: Path, *, expected_rows: int) -> bool:
    if expected_rows <= 0:
        return False
    return not path.exists() or line_count(path) < expected_rows


def pending_response_entry(
    name: str,
    *,
    jobs_path: Path,
    responses_path: Path,
    expected_rows: int,
) -> dict[str, Any]:
    existing_rows = line_count(responses_path) if responses_path.exists() else 0
    remaining_rows = max(0, expected_rows - existing_rows)
    return {
        "name": name,
        "jobs_jsonl": str(jobs_path),
        "responses_jsonl": str(responses_path),
        "expected_rows": expected_rows,
        "existing_rows": existing_rows,
        "remaining_rows": remaining_rows,
        "runner_template": (
            "python training/run_curriculum_job_responses.py "
            f"--jobs_jsonl {jobs_path} "
            f"--output_jsonl {responses_path} "
            "--backend command "
            "--command \"python scripts/provider_runner.py\" "
            "--resume --fail_fast"
        ),
    }


def validate_role_models(
    role_name: str,
    models: list[str],
    *,
    allow_single_model_roles: bool,
    allow_student_lineage: bool,
) -> None:
    validate_external_models(models, allow_student_lineage=allow_student_lineage)
    distinct = {model.strip().lower() for model in models if model.strip()}
    if not allow_single_model_roles and len(distinct) < 2:
        raise ValueError(
            f"{role_name} requires at least two distinct external models. "
            "Use --allow_single_model_roles only for local smoke tests."
        )


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    paths = artifact_paths(work_dir)
    references = existing_reference_paths(args.references_jsonl or [], ROOT)
    if not references:
        raise ValueError("No reference JSONL files found for decontamination.")

    seed_models = split_csv(args.seed_models)
    solver_models = split_csv(args.solver_models)
    judge_models = split_csv(args.judge_models)
    methods = split_csv(args.methods)
    if not (seed_models and solver_models and judge_models):
        raise ValueError("seed, solver, and judge models are required.")
    validate_role_models(
        "seed_models",
        seed_models,
        allow_single_model_roles=args.allow_single_model_roles,
        allow_student_lineage=args.allow_student_lineage,
    )
    validate_role_models(
        "solver_models",
        solver_models,
        allow_single_model_roles=args.allow_single_model_roles,
        allow_student_lineage=args.allow_student_lineage,
    )
    validate_role_models(
        "judge_models",
        judge_models,
        allow_single_model_roles=args.allow_single_model_roles,
        allow_student_lineage=args.allow_student_lineage,
    )

    seed_jobs = build_seed_jobs(
        models=seed_models,
        domains=split_csv(args.domains) or ["math"],
        difficulties=split_csv(args.difficulties) or ["medium"],
        target_steps=args.target_steps,
        count_per_combo=args.count_per_combo,
    )
    write_jsonl(paths["jobs_seed"], seed_jobs)
    if response_incomplete(paths["responses_seed"], expected_rows=len(seed_jobs)):
        return stop_summary(
            work_dir=work_dir,
            status="pending_seed_responses",
            next_action=f"Run provider responses for {paths['jobs_seed']}",
            artifacts=paths,
            counts={
                "seed_jobs": len(seed_jobs),
                "seed_response_rows": line_count(paths["responses_seed"]) if paths["responses_seed"].exists() else 0,
            },
            pending_responses=[
                pending_response_entry(
                    "seed",
                    jobs_path=paths["jobs_seed"],
                    responses_path=paths["responses_seed"],
                    expected_rows=len(seed_jobs),
                )
            ],
        )

    candidates, candidates_report = seed_outputs_to_candidates(seed_jobs, read_jsonl(paths["responses_seed"]))
    write_jsonl(paths["candidates"], candidates)
    write_json(paths["candidates_report"], candidates_report)
    reference_index = build_reference_index(
        references,
        text_fields=("statement", "prompt", "question", "problem", "input", "text"),
        ngram_size=args.decontam_ngram_size,
        min_ngram_size=args.decontam_min_ngram_size,
    )
    decontaminated, contaminated, decontam_report = annotate_candidates(
        candidates,
        reference_index,
        text_fields=("statement", "prompt", "question", "problem", "input", "text"),
        ngram_size=args.decontam_ngram_size,
        min_ngram_size=args.decontam_min_ngram_size,
        threshold=args.decontam_threshold,
    )
    decontam_report.pop("annotated_rows", None)
    write_jsonl(paths["candidates_decontaminated"], decontaminated)
    write_jsonl(paths["candidates_contaminated"], contaminated)
    write_json(paths["decontam_report"], decontam_report)

    ground_jobs = build_ground_truth_jobs(decontaminated, models=solver_models)
    write_jsonl(paths["jobs_ground_truth"], ground_jobs)
    if response_incomplete(paths["responses_ground_truth"], expected_rows=len(ground_jobs)):
        return stop_summary(
            work_dir=work_dir,
            status="pending_ground_truth_responses",
            next_action=f"Run provider responses for {paths['jobs_ground_truth']}",
            artifacts=paths,
            counts={
                "seed_candidates": candidates_report["candidates"],
                "decontaminated": decontam_report["accepted"],
                "contaminated": decontam_report["rejected"],
                "ground_truth_jobs": len(ground_jobs),
                "ground_truth_response_rows": line_count(paths["responses_ground_truth"])
                if paths["responses_ground_truth"].exists()
                else 0,
            },
            pending_responses=[
                pending_response_entry(
                    "ground_truth",
                    jobs_path=paths["jobs_ground_truth"],
                    responses_path=paths["responses_ground_truth"],
                    expected_rows=len(ground_jobs),
                )
            ],
        )

    verified, verified_report = ground_truth_outputs_to_verified_candidates(
        decontaminated,
        ground_jobs,
        read_jsonl(paths["responses_ground_truth"]),
        min_agree=args.min_ground_truth_agree,
        require_claimed_answer_match=args.require_claimed_answer_match,
        require_programmatic_answer_check=args.require_programmatic_answer_check,
    )
    write_jsonl(paths["verified_candidates"], verified)
    write_json(paths["verified_candidates_report"], verified_report)

    reference_jobs = build_reference_attempt_jobs(
        verified,
        model=args.reference_model,
        samples=args.reference_samples,
    )
    write_jsonl(paths["jobs_reference_attempts"], reference_jobs)
    if response_missing(paths["reference_attempts"]):
        if response_incomplete(paths["responses_reference_attempts"], expected_rows=len(reference_jobs)):
            return stop_summary(
                work_dir=work_dir,
                status="pending_reference_attempt_responses",
                next_action=f"Run provider responses for {paths['jobs_reference_attempts']}",
                artifacts=paths,
                counts={
                    "verified": verified_report["verified"],
                    "ground_truth_rejected": verified_report["rejected"],
                    "reference_attempt_jobs": len(reference_jobs),
                    "reference_attempt_response_rows": line_count(paths["responses_reference_attempts"])
                    if paths["responses_reference_attempts"].exists()
                    else 0,
                },
                pending_responses=[
                    pending_response_entry(
                        "reference_attempt",
                        jobs_path=paths["jobs_reference_attempts"],
                        responses_path=paths["responses_reference_attempts"],
                        expected_rows=len(reference_jobs),
                    )
                ],
            )
        reference_attempts, reference_attempts_report = reference_outputs_to_attempts(
            verified,
            reference_jobs,
            read_jsonl(paths["responses_reference_attempts"]),
        )
        write_jsonl(paths["reference_attempts"], reference_attempts)
        write_json(paths["reference_attempts_report"], reference_attempts_report)
    else:
        reference_attempts = read_jsonl(paths["reference_attempts"])
        reference_attempts_report = {
            "mode": "reference_attempts_existing",
            "attempts": len(reference_attempts),
            "issues": [],
        }
        write_json(paths["reference_attempts_report"], reference_attempts_report)

    if response_missing(paths["reference_attempts"]):
        return stop_summary(
            work_dir=work_dir,
            status="pending_reference_attempts",
            next_action=f"Write weak-reference attempts to {paths['reference_attempts']}",
            artifacts=paths,
            counts={
                "verified": verified_report["verified"],
                "ground_truth_rejected": verified_report["rejected"],
            },
        )

    with_difficulty, difficulty_rejected, difficulty_report = annotate_difficulty(
        verified,
        reference_attempts,
        reference_model=args.reference_model,
        min_samples=args.min_reference_samples,
        drop_unmeasured=args.drop_unmeasured_difficulty,
    )
    write_jsonl(paths["verified_candidates_difficulty"], with_difficulty)
    write_json(paths["difficulty_report"], difficulty_report)
    with_false, false_rejected, false_report = annotate_false_answers(
        with_difficulty,
        drop_unannotated=args.drop_unannotated_false_answers,
    )
    write_jsonl(paths["verified_candidates_no_false_answer"], false_rejected)
    write_jsonl(paths["verified_candidates_false_answers"], with_false)
    write_json(paths["false_answers_report"], false_report)

    method_jobs = build_method_solve_jobs(with_false, models=solver_models, fallback_methods=methods)
    perturbation_jobs = build_perturbation_jobs(with_false, models=solver_models)
    write_jsonl(paths["jobs_methods"], method_jobs)
    write_jsonl(paths["jobs_perturbation"], perturbation_jobs)
    if response_incomplete(paths["responses_methods"], expected_rows=len(method_jobs)) or response_incomplete(
        paths["responses_perturbation"],
        expected_rows=len(perturbation_jobs),
    ):
        return stop_summary(
            work_dir=work_dir,
            status="pending_method_or_perturbation_responses",
            next_action=f"Run provider responses for {paths['jobs_methods']} and {paths['jobs_perturbation']}",
            artifacts=paths,
            counts={
                "verified_with_difficulty": difficulty_report["annotated"],
                "reference_attempts": reference_attempts_report["attempts"],
                "false_answers": false_report["with_false_answer"],
                "method_jobs": len(method_jobs),
                "perturbation_jobs": len(perturbation_jobs),
                "method_response_rows": line_count(paths["responses_methods"]) if paths["responses_methods"].exists() else 0,
                "perturbation_response_rows": line_count(paths["responses_perturbation"])
                if paths["responses_perturbation"].exists()
                else 0,
            },
            pending_responses=[
                entry
                for entry in [
                    pending_response_entry(
                        "method_solve",
                        jobs_path=paths["jobs_methods"],
                        responses_path=paths["responses_methods"],
                        expected_rows=len(method_jobs),
                    )
                    if response_incomplete(paths["responses_methods"], expected_rows=len(method_jobs))
                    else None,
                    pending_response_entry(
                        "perturbation",
                        jobs_path=paths["jobs_perturbation"],
                        responses_path=paths["responses_perturbation"],
                        expected_rows=len(perturbation_jobs),
                    )
                    if response_incomplete(paths["responses_perturbation"], expected_rows=len(perturbation_jobs))
                    else None,
                ]
                if entry is not None
            ],
        )

    method_solutions, method_report = method_outputs_to_solution_candidates(
        with_false,
        method_jobs,
        read_jsonl(paths["responses_methods"]),
    )
    write_jsonl(paths["method_solution_candidates"], method_solutions)
    write_json(paths["method_solutions_report"], method_report)
    perturbation_traces, perturbation_report = perturbation_outputs_to_traces(
        with_false,
        perturbation_jobs,
        read_jsonl(paths["responses_perturbation"]) if perturbation_jobs else [],
    )
    write_jsonl(paths["perturbation_traces"], perturbation_traces)
    write_json(paths["perturbation_report"], perturbation_report)

    naturalness_jobs = build_naturalness_jobs(method_solutions, models=judge_models)
    distinctness_jobs = build_distinctness_jobs(method_solutions, models=judge_models)
    depth_jobs = build_depth_jobs(method_solutions, models=judge_models)
    error_detection_jobs = build_error_detection_jobs(perturbation_traces, models=judge_models)
    write_jsonl(paths["jobs_naturalness"], naturalness_jobs)
    write_jsonl(paths["jobs_distinctness"], distinctness_jobs)
    write_jsonl(paths["jobs_depth"], depth_jobs)
    write_jsonl(paths["jobs_error_detection"], error_detection_jobs)
    required_judgment_missing = (
        response_incomplete(paths["responses_naturalness"], expected_rows=len(naturalness_jobs))
        or response_incomplete(paths["responses_distinctness"], expected_rows=len(distinctness_jobs))
        or response_incomplete(paths["responses_depth"], expected_rows=len(depth_jobs))
    )
    optional_error_missing = response_incomplete(
        paths["responses_error_detection"],
        expected_rows=len(error_detection_jobs),
    )
    if required_judgment_missing or (args.require_error_detection and optional_error_missing):
        return stop_summary(
            work_dir=work_dir,
            status="pending_judgment_responses",
            next_action=(
                f"Run provider responses for {paths['jobs_naturalness']}, "
                f"{paths['jobs_distinctness']}, {paths['jobs_depth']}"
                + (f", and {paths['jobs_error_detection']}" if args.require_error_detection else "")
            ),
            artifacts=paths,
            counts={
                "method_solutions": method_report["solution_candidates"],
                "perturbation_traces": perturbation_report["traces"],
                "naturalness_jobs": len(naturalness_jobs),
                "distinctness_jobs": len(distinctness_jobs),
                "depth_jobs": len(depth_jobs),
                "error_detection_jobs": len(error_detection_jobs),
                "naturalness_response_rows": line_count(paths["responses_naturalness"])
                if paths["responses_naturalness"].exists()
                else 0,
                "distinctness_response_rows": line_count(paths["responses_distinctness"])
                if paths["responses_distinctness"].exists()
                else 0,
                "depth_response_rows": line_count(paths["responses_depth"]) if paths["responses_depth"].exists() else 0,
                "error_detection_response_rows": line_count(paths["responses_error_detection"])
                if paths["responses_error_detection"].exists()
                else 0,
            },
            pending_responses=[
                entry
                for entry in [
                    pending_response_entry(
                        "naturalness",
                        jobs_path=paths["jobs_naturalness"],
                        responses_path=paths["responses_naturalness"],
                        expected_rows=len(naturalness_jobs),
                    )
                    if response_incomplete(paths["responses_naturalness"], expected_rows=len(naturalness_jobs))
                    else None,
                    pending_response_entry(
                        "distinctness",
                        jobs_path=paths["jobs_distinctness"],
                        responses_path=paths["responses_distinctness"],
                        expected_rows=len(distinctness_jobs),
                    )
                    if response_incomplete(paths["responses_distinctness"], expected_rows=len(distinctness_jobs))
                    else None,
                    pending_response_entry(
                        "depth",
                        jobs_path=paths["jobs_depth"],
                        responses_path=paths["responses_depth"],
                        expected_rows=len(depth_jobs),
                    )
                    if response_incomplete(paths["responses_depth"], expected_rows=len(depth_jobs))
                    else None,
                    pending_response_entry(
                        "error_detection",
                        jobs_path=paths["jobs_error_detection"],
                        responses_path=paths["responses_error_detection"],
                        expected_rows=len(error_detection_jobs),
                    )
                    if args.require_error_detection
                    and response_incomplete(paths["responses_error_detection"], expected_rows=len(error_detection_jobs))
                    else None,
                ]
                if entry is not None
            ],
        )

    naturalness, naturalness_report = collect_naturalness_judgments(
        method_solutions,
        naturalness_jobs,
        read_jsonl(paths["responses_naturalness"]),
    )
    distinctness, distinctness_report = collect_distinctness_judgments(
        method_solutions,
        distinctness_jobs,
        read_jsonl(paths["responses_distinctness"]),
    )
    depth, depth_report = collect_depth_measurements(method_solutions, depth_jobs, read_jsonl(paths["responses_depth"]))
    write_jsonl(paths["naturalness_judgments"], naturalness)
    write_jsonl(paths["distinctness_judgments"], distinctness)
    write_jsonl(paths["depth_measurements"], depth)
    write_json(paths["naturalness_report"], naturalness_report)
    write_json(paths["distinctness_report"], distinctness_report)
    write_json(paths["depth_report"], depth_report)
    error_detection: list[dict[str, Any]] = []
    error_report: dict[str, Any] = {"judgments": 0, "issues": [], "status_counts": {}}
    if error_detection_jobs and paths["responses_error_detection"].exists() and line_count(paths["responses_error_detection"]) > 0:
        error_detection, error_report = collect_error_detection_judgments(
            perturbation_traces,
            error_detection_jobs,
            read_jsonl(paths["responses_error_detection"]),
        )
        write_jsonl(paths["error_detection_judgments"], error_detection)
    write_json(paths["error_detection_report"], error_report)

    records, records_report = assemble_curriculum_records(
        with_false,
        method_solutions,
        naturalness,
        depth,
        distinctness,
        [*perturbation_traces, *error_detection],
        min_natural_agree=args.min_natural_agree,
        min_distinct_agree=args.min_distinct_agree,
        deep_threshold=args.deep_threshold,
    )
    write_jsonl(paths["typed_records"], records)
    write_json(paths["typed_records_report"], records_report)
    sft_rows, sft_report = convert_curriculum_records(records)
    write_jsonl(paths["positive_sft"], sft_rows)
    write_json(paths["positive_sft_report"], sft_report)

    return stop_summary(
        work_dir=work_dir,
        status="complete",
        next_action="Review typed_records.jsonl and positive_sft.jsonl before any GPU fine-tuning.",
        artifacts=paths,
        counts={
            "seed_candidates": candidates_report["candidates"],
            "decontaminated": decontam_report["accepted"],
            "verified": verified_report["verified"],
            "reference_attempts": reference_attempts_report["attempts"],
            "difficulty_measured": difficulty_report["measured"],
            "false_answers": false_report["with_false_answer"],
            "method_solutions": method_report["solution_candidates"],
            "perturbation_traces": perturbation_report["traces"],
            "naturalness_judgments": naturalness_report["judgments"],
            "distinctness_judgments": distinctness_report["judgments"],
            "depth_measurements": depth_report["measurements"],
            "error_detection_judgments": error_report["judgments"],
            "typed_records": records_report["records"],
            "positive_sft_rows": sft_report["exported_examples"],
            "mode_counts": records_report["mode_counts"],
        },
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work_dir", default="data/curriculum/run")
    parser.add_argument("--seed_models", default="opus-strong,glm-strong")
    parser.add_argument("--solver_models", default="opus-strong,glm-strong")
    parser.add_argument("--judge_models", default="opus-strong,glm-strong")
    parser.add_argument("--allow_single_model_roles", action="store_true")
    parser.add_argument("--allow_student_lineage", action="store_true")
    parser.add_argument("--domains", default="math")
    parser.add_argument("--difficulties", default="medium,hard")
    parser.add_argument("--target_steps", type=parse_int_csv, default=[4, 8])
    parser.add_argument("--count_per_combo", type=int, default=1)
    parser.add_argument("--methods", default="algebra,number_theory,bounded_enumeration")
    parser.add_argument("--references_jsonl", action="append")
    parser.add_argument("--decontam_ngram_size", type=int, default=5)
    parser.add_argument("--decontam_min_ngram_size", type=int, default=3)
    parser.add_argument("--decontam_threshold", type=float, default=0.5)
    parser.add_argument("--min_ground_truth_agree", type=int, default=2)
    parser.add_argument("--require_claimed_answer_match", action="store_true")
    parser.add_argument(
        "--require_programmatic_answer_check",
        action="store_true",
        help=(
            "Reject generated candidates unless the agreed solver answer passes a cheap deterministic "
            "check against the seed claimed_answer. Use for strong-model generated shards before SFT."
        ),
    )
    parser.add_argument("--reference_model", default="weak-reference")
    parser.add_argument("--reference_samples", type=int, default=3)
    parser.add_argument("--min_reference_samples", type=int, default=1)
    parser.add_argument("--drop_unmeasured_difficulty", action="store_true")
    parser.add_argument("--drop_unannotated_false_answers", action="store_true")
    parser.add_argument("--min_natural_agree", type=int, default=2)
    parser.add_argument("--min_distinct_agree", type=int, default=2)
    parser.add_argument("--deep_threshold", type=int, default=5)
    parser.add_argument("--require_error_detection", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    summary = run_pipeline(parse_args(argv))
    print(f"status={summary['status']}")
    print(f"next_action={summary['next_action']}")
    print(f"summary_json={summary['artifacts']['summary']['path']}")
    print(json.dumps(summary["counts"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
