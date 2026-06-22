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
    build_seed_jobs,
)
from training.collect_curriculum_job_outputs import (
    collect_depth_measurements,
    collect_distinctness_judgments,
    collect_error_detection_judgments,
    collect_naturalness_judgments,
    ground_truth_outputs_to_verified_candidates,
    method_outputs_to_solution_candidates,
    perturbation_outputs_to_traces,
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
        "candidates_decontaminated": work_dir / "candidates_decontaminated.jsonl",
        "candidates_contaminated": work_dir / "candidates_contaminated.jsonl",
        "jobs_ground_truth": work_dir / "jobs_ground_truth.jsonl",
        "responses_ground_truth": work_dir / "responses_ground_truth.jsonl",
        "verified_candidates": work_dir / "verified_candidates.jsonl",
        "reference_attempts": work_dir / "reference_attempts.jsonl",
        "verified_candidates_difficulty": work_dir / "verified_candidates_difficulty.jsonl",
        "verified_candidates_no_false_answer": work_dir / "verified_candidates_no_false_answer.jsonl",
        "verified_candidates_false_answers": work_dir / "verified_candidates_false_answers.jsonl",
        "jobs_methods": work_dir / "jobs_methods.jsonl",
        "responses_methods": work_dir / "responses_methods.jsonl",
        "method_solution_candidates": work_dir / "method_solution_candidates.jsonl",
        "jobs_perturbation": work_dir / "jobs_perturbation.jsonl",
        "responses_perturbation": work_dir / "responses_perturbation.jsonl",
        "perturbation_traces": work_dir / "perturbation_traces.jsonl",
        "jobs_naturalness": work_dir / "jobs_naturalness.jsonl",
        "responses_naturalness": work_dir / "responses_naturalness.jsonl",
        "naturalness_judgments": work_dir / "naturalness_judgments.jsonl",
        "jobs_distinctness": work_dir / "jobs_distinctness.jsonl",
        "responses_distinctness": work_dir / "responses_distinctness.jsonl",
        "distinctness_judgments": work_dir / "distinctness_judgments.jsonl",
        "jobs_depth": work_dir / "jobs_depth.jsonl",
        "responses_depth": work_dir / "responses_depth.jsonl",
        "depth_measurements": work_dir / "depth_measurements.jsonl",
        "jobs_error_detection": work_dir / "jobs_error_detection.jsonl",
        "responses_error_detection": work_dir / "responses_error_detection.jsonl",
        "error_detection_judgments": work_dir / "error_detection_judgments.jsonl",
        "typed_records": work_dir / "typed_records.jsonl",
        "positive_sft": work_dir / "positive_sft.jsonl",
        "summary": work_dir / "summary.json",
    }


def stop_summary(
    *,
    work_dir: Path,
    status: str,
    next_action: str,
    artifacts: dict[str, Path],
    counts: dict[str, Any] | None = None,
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
    write_json(artifacts["summary"], summary)
    return summary


def line_count(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def response_missing(path: Path) -> bool:
    return not path.exists() or line_count(path) == 0


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

    seed_jobs = build_seed_jobs(
        models=seed_models,
        domains=split_csv(args.domains) or ["math"],
        difficulties=split_csv(args.difficulties) or ["medium"],
        target_steps=args.target_steps,
        count_per_combo=args.count_per_combo,
    )
    write_jsonl(paths["jobs_seed"], seed_jobs)
    if response_missing(paths["responses_seed"]):
        return stop_summary(
            work_dir=work_dir,
            status="pending_seed_responses",
            next_action=f"Run provider responses for {paths['jobs_seed']}",
            artifacts=paths,
            counts={"seed_jobs": len(seed_jobs)},
        )

    candidates, candidates_report = seed_outputs_to_candidates(seed_jobs, read_jsonl(paths["responses_seed"]))
    write_jsonl(paths["candidates"], candidates)
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

    ground_jobs = build_ground_truth_jobs(decontaminated, models=solver_models)
    write_jsonl(paths["jobs_ground_truth"], ground_jobs)
    if response_missing(paths["responses_ground_truth"]):
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
            },
        )

    verified, verified_report = ground_truth_outputs_to_verified_candidates(
        decontaminated,
        ground_jobs,
        read_jsonl(paths["responses_ground_truth"]),
        min_agree=args.min_ground_truth_agree,
    )
    write_jsonl(paths["verified_candidates"], verified)
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
        read_jsonl(paths["reference_attempts"]),
        reference_model=args.reference_model,
        min_samples=args.min_reference_samples,
        drop_unmeasured=args.drop_unmeasured_difficulty,
    )
    write_jsonl(paths["verified_candidates_difficulty"], with_difficulty)
    with_false, false_rejected, false_report = annotate_false_answers(
        with_difficulty,
        drop_unannotated=args.drop_unannotated_false_answers,
    )
    write_jsonl(paths["verified_candidates_no_false_answer"], false_rejected)
    write_jsonl(paths["verified_candidates_false_answers"], with_false)

    method_jobs = build_method_solve_jobs(with_false, models=solver_models, fallback_methods=methods)
    perturbation_jobs = build_perturbation_jobs(with_false, models=solver_models)
    write_jsonl(paths["jobs_methods"], method_jobs)
    write_jsonl(paths["jobs_perturbation"], perturbation_jobs)
    if response_missing(paths["responses_methods"]) or (perturbation_jobs and response_missing(paths["responses_perturbation"])):
        return stop_summary(
            work_dir=work_dir,
            status="pending_method_or_perturbation_responses",
            next_action=f"Run provider responses for {paths['jobs_methods']} and {paths['jobs_perturbation']}",
            artifacts=paths,
            counts={
                "verified_with_difficulty": difficulty_report["annotated"],
                "false_answers": false_report["with_false_answer"],
                "method_jobs": len(method_jobs),
                "perturbation_jobs": len(perturbation_jobs),
            },
        )

    method_solutions, method_report = method_outputs_to_solution_candidates(
        with_false,
        method_jobs,
        read_jsonl(paths["responses_methods"]),
    )
    write_jsonl(paths["method_solution_candidates"], method_solutions)
    perturbation_traces, perturbation_report = perturbation_outputs_to_traces(
        with_false,
        perturbation_jobs,
        read_jsonl(paths["responses_perturbation"]) if perturbation_jobs else [],
    )
    write_jsonl(paths["perturbation_traces"], perturbation_traces)

    naturalness_jobs = build_naturalness_jobs(method_solutions, models=judge_models)
    distinctness_jobs = build_distinctness_jobs(method_solutions, models=judge_models)
    depth_jobs = build_depth_jobs(method_solutions, models=judge_models)
    error_detection_jobs = build_error_detection_jobs(perturbation_traces, models=judge_models)
    write_jsonl(paths["jobs_naturalness"], naturalness_jobs)
    write_jsonl(paths["jobs_distinctness"], distinctness_jobs)
    write_jsonl(paths["jobs_depth"], depth_jobs)
    write_jsonl(paths["jobs_error_detection"], error_detection_jobs)
    required_judgment_missing = (
        response_missing(paths["responses_naturalness"])
        or response_missing(paths["responses_distinctness"])
        or response_missing(paths["responses_depth"])
    )
    optional_error_missing = error_detection_jobs and response_missing(paths["responses_error_detection"])
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
            },
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
    error_detection: list[dict[str, Any]] = []
    error_report: dict[str, Any] = {"judgments": 0, "issues": [], "status_counts": {}}
    if error_detection_jobs and paths["responses_error_detection"].exists() and line_count(paths["responses_error_detection"]) > 0:
        error_detection, error_report = collect_error_detection_judgments(
            perturbation_traces,
            error_detection_jobs,
            read_jsonl(paths["responses_error_detection"]),
        )
        write_jsonl(paths["error_detection_judgments"], error_detection)

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
    sft_rows, sft_report = convert_curriculum_records(records)
    write_jsonl(paths["positive_sft"], sft_rows)

    return stop_summary(
        work_dir=work_dir,
        status="complete",
        next_action="Review typed_records.jsonl and positive_sft.jsonl before any GPU fine-tuning.",
        artifacts=paths,
        counts={
            "seed_candidates": candidates_report["candidates"],
            "decontaminated": decontam_report["accepted"],
            "verified": verified_report["verified"],
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
    parser.add_argument("--reference_model", default="weak-reference")
    parser.add_argument("--min_reference_samples", type=int, default=1)
    parser.add_argument("--drop_unmeasured_difficulty", action="store_true")
    parser.add_argument("--drop_unannotated_false_answers", action="store_true")
    parser.add_argument("--min_natural_agree", type=int, default=1)
    parser.add_argument("--min_distinct_agree", type=int, default=1)
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
