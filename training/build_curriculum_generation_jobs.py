"""Build provider-neutral strong-model curriculum generation jobs.

This script does not call any model API. It creates JSONL job records for the
wide/deep curriculum pipeline so external runners can submit prompts to strong
non-student models and write raw responses back for later verification.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


STUDENT_LINEAGE_PATTERNS = ("qwen",)

MATH_METHODS: dict[str, str] = {
    "algebra": "algebraic manipulation of equations, expressions, or identities",
    "coordinate_geometry": "coordinate representation of geometric objects",
    "synthetic_geometry": "diagrammatic or theorem-based geometry without coordinates",
    "trigonometry": "trigonometric ratios, identities, or angle relations",
    "induction": "proof by base case and inductive step",
    "number_theory": "modular arithmetic, divisibility, or prime-factor structure",
    "combinatorial_argument": "counting, bijection, inclusion-exclusion, or constructive combinatorics",
    "generating_functions": "generating functions or coefficient extraction",
    "inequalities": "classical inequalities such as AM-GM, Cauchy-Schwarz, or Jensen",
    "extremal_pigeonhole": "extremal argument or pigeonhole principle",
    "complex_numbers": "complex-plane representation or algebra",
    "calculus_limits": "derivatives, integrals, or limiting arguments",
    "bounded_enumeration": "small exhaustive enumeration with explicit constraints",
}

CODE_METHODS: dict[str, str] = {
    "iterative": "loop-based direct computation",
    "recursive": "recursive decomposition",
    "dynamic_programming": "memoized or tabulated overlapping subproblems",
    "greedy": "locally optimal choices with a correctness argument",
    "divide_and_conquer": "split, solve subproblems, then combine",
    "graph_search": "graph, tree, BFS, DFS, shortest path, or state-space search",
    "closed_form": "mathematical simplification or closed-form solution",
    "hashing_sets": "hash tables, sets, or membership/inversion structure",
}


def split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def slug(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return value or "item"


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"Line {line_no} is not a JSON object.")
        rows.append(row)
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def method_taxonomy(domain: str) -> dict[str, str]:
    domain = domain.strip().lower()
    if domain in {"math", "mathematics", "stem"}:
        return MATH_METHODS
    if domain in {"code", "coding", "programming"}:
        return CODE_METHODS
    return {}


def validate_external_models(models: list[str], *, allow_student_lineage: bool = False) -> None:
    if allow_student_lineage:
        return
    blocked = [
        model
        for model in models
        if any(pattern in model.lower() for pattern in STUDENT_LINEAGE_PATTERNS)
    ]
    if blocked:
        raise ValueError(
            "Student-lineage models are blocked for curriculum generation unless "
            f"--allow_student_lineage is set: {blocked}"
        )


def prompt_problem_generation(*, domain: str, difficulty: str, target_steps: int) -> str:
    return (
        f"Generate one {difficulty} {domain} problem for a reasoning dataset.\n"
        "Requirements:\n"
        "- It must have a single, unambiguous final answer that can be verified: "
        "a number, a closed-form expression, or, for code, a precise function "
        "specification with input and output types.\n"
        "- State the problem precisely. Do NOT include a solution.\n"
        f"- Calibrate difficulty so a competent non-expert needs roughly {target_steps} steps.\n"
        "Return ONLY this JSON:\n"
        "{\"statement\": \"...\", \"claimed_answer\": \"...\", "
        f"\"domain\": \"{domain}\", \"candidate_methods\": [\"...\", \"...\"]}}\n"
        "The claimed_answer will be independently verified, so solve carefully rather than guessing."
    )


def prompt_independent_solve(statement: str) -> str:
    return (
        "Solve the problem. Reason rigorously and do not assume any externally provided answer.\n"
        "End with the final answer on its own line beginning \"ANSWER:\".\n"
        f"Problem: {statement}"
    )


def prompt_method_solve(*, statement: str, method: str, method_description: str) -> str:
    return (
        f"Solve the problem using ONLY the method: {method} ({method_description}).\n"
        "Rules:\n"
        f"- {method} must be the essential technique. Do not switch to a different method partway.\n"
        "- If the method does not naturally apply here, do NOT force it. Reply with exactly:\n"
        "  METHOD DOES NOT APPLY\n"
        "  followed by one sentence on why.\n"
        "If it applies, give a clean, natural solution by the method, then the final answer "
        "on its own line beginning \"ANSWER:\".\n"
        f"Problem: {statement}"
    )


def prompt_naturalness_judge(*, statement: str, solution: str, method: str) -> str:
    return (
        f"A solution below claims to use {method}. Judge whether it is a genuine and natural "
        "use of that method, or forced or degenerate.\n"
        "Return ONLY this JSON:\n"
        "{\"natural\": true, \"actually_uses\": \"method_or_real_method\", \"reason\": \"...\"}\n"
        f"Problem: {statement}\n"
        f"Solution: {solution}"
    )


def prompt_depth_decomposition(solution: str) -> str:
    return (
        "Break the correct solution below into its minimal sequence of necessary steps, "
        "where each step is a single inference or computation.\n"
        "Return ONLY this JSON: {\"steps\": [\"...\", \"...\"], \"count\": 2}\n"
        f"Solution: {solution}"
    )


def prompt_false_answer_neutral(*, statement: str, false_answer: str) -> str:
    return (
        f"A student claims the answer to this problem is {false_answer}. "
        "Solve the problem yourself and state whether the student is correct.\n"
        "End with your final answer on its own line beginning \"ANSWER:\".\n"
        f"Problem: {statement}"
    )


def prompt_false_answer_pressure(*, statement: str, false_answer: str) -> str:
    return (
        f"The answer to this problem is {false_answer}. Show the full reasoning that arrives at it.\n"
        f"Problem: {statement}"
    )


def prompt_error_detection(*, statement: str, solution: str) -> str:
    return (
        "A problem and a proposed solution are below. The solution may be correct or may contain an error.\n"
        "Decide. If incorrect, identify the FIRST step where the reasoning goes wrong and explain.\n"
        "Return ONLY this JSON:\n"
        "{\"verdict\": \"correct\", \"first_error_step\": null, \"explanation\": \"...\"}\n"
        f"Problem: {statement}\n"
        f"Proposed solution: {solution}"
    )


def make_job(
    *,
    index: int,
    stage: str,
    role: str,
    model: str,
    prompt: str,
    expects_json: bool,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "job_id": f"{stage}-{index:06d}-{slug(role)}-{slug(model)}",
        "stage": stage,
        "role": role,
        "model": model,
        "prompt": prompt,
        "expects_json": expects_json,
        "metadata": metadata,
        "routing": {
            "trusted_for": "generation_only",
            "requires_independent_verification": True,
        },
    }


def record_id(row: dict[str, Any], index: int) -> str:
    return str(row.get("id") or row.get("problem_id") or f"row-{index:06d}")


def statement_for(row: dict[str, Any]) -> str:
    return str(row.get("statement") or row.get("prompt") or "").strip()


def solution_for(row: dict[str, Any]) -> str:
    return str(row.get("solution") or row.get("text") or row.get("trace") or row.get("completion") or "").strip()


def explicit_methods(row: dict[str, Any]) -> list[str]:
    if isinstance(row.get("candidate_methods"), list):
        return [str(item) for item in row["candidate_methods"] if str(item).strip()]
    width_signature = row.get("width_signature")
    if isinstance(width_signature, dict) and isinstance(width_signature.get("methods"), list):
        return [str(item) for item in width_signature["methods"] if str(item).strip()]
    return []


def build_seed_jobs(
    *,
    models: list[str],
    domains: list[str],
    difficulties: list[str],
    target_steps: list[int],
    count_per_combo: int,
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for domain in domains:
        for difficulty in difficulties:
            for steps in target_steps:
                for copy_index in range(count_per_combo):
                    for model in models:
                        prompt = prompt_problem_generation(
                            domain=domain,
                            difficulty=difficulty,
                            target_steps=steps,
                        )
                        jobs.append(
                            make_job(
                                index=len(jobs),
                                stage="seed_generation",
                                role="generator",
                                model=model,
                                prompt=prompt,
                                expects_json=True,
                                metadata={
                                    "domain": domain,
                                    "difficulty": difficulty,
                                    "target_steps": steps,
                                    "copy_index": copy_index,
                                },
                            )
                        )
    return jobs


def build_ground_truth_jobs(rows: list[dict[str, Any]], *, models: list[str]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        statement = statement_for(row)
        if not statement:
            continue
        for model in models:
            jobs.append(
                make_job(
                    index=len(jobs),
                    stage="ground_truth_solve",
                    role="solver",
                    model=model,
                    prompt=prompt_independent_solve(statement),
                    expects_json=False,
                    metadata={"record_id": record_id(row, row_index), "domain": row.get("domain")},
                )
            )
    return jobs


def methods_for_row(row: dict[str, Any], *, fallback_methods: list[str]) -> dict[str, str]:
    domain_methods = method_taxonomy(str(row.get("domain") or "math"))
    selected = explicit_methods(row) or fallback_methods or list(domain_methods)
    return {
        method: domain_methods.get(method, method.replace("_", " "))
        for method in selected
        if str(method).strip()
    }


def build_method_solve_jobs(
    rows: list[dict[str, Any]],
    *,
    models: list[str],
    fallback_methods: list[str],
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        statement = statement_for(row)
        if not statement:
            continue
        for method, description in methods_for_row(row, fallback_methods=fallback_methods).items():
            for model in models:
                jobs.append(
                    make_job(
                        index=len(jobs),
                        stage="method_constrained_solve",
                        role="method_solver",
                        model=model,
                        prompt=prompt_method_solve(
                            statement=statement,
                            method=method,
                            method_description=description,
                        ),
                        expects_json=False,
                        metadata={
                            "record_id": record_id(row, row_index),
                            "domain": row.get("domain"),
                            "method": method,
                        },
                    )
                )
    return jobs


def build_naturalness_jobs(rows: list[dict[str, Any]], *, models: list[str]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        statement = statement_for(row)
        solution = solution_for(row)
        method = str(row.get("method") or "").strip()
        if not (statement and solution and method):
            continue
        for model in models:
            jobs.append(
                make_job(
                    index=len(jobs),
                    stage="naturalness_judge",
                    role="judge",
                    model=model,
                    prompt=prompt_naturalness_judge(statement=statement, solution=solution, method=method),
                    expects_json=True,
                    metadata={"record_id": record_id(row, row_index), "method": method},
                )
            )
    return jobs


def build_depth_jobs(rows: list[dict[str, Any]], *, models: list[str]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        solution = solution_for(row)
        if not solution:
            continue
        for model in models:
            jobs.append(
                make_job(
                    index=len(jobs),
                    stage="depth_decomposition",
                    role="judge",
                    model=model,
                    prompt=prompt_depth_decomposition(solution),
                    expects_json=True,
                    metadata={"record_id": record_id(row, row_index), "method": row.get("method")},
                )
            )
    return jobs


def build_perturbation_jobs(rows: list[dict[str, Any]], *, models: list[str]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        statement = statement_for(row)
        false_answer = str(row.get("false_answer") or "").strip()
        if not (statement and false_answer):
            continue
        for model in models:
            common = {"record_id": record_id(row, row_index), "false_answer": false_answer}
            jobs.append(
                make_job(
                    index=len(jobs),
                    stage="false_answer_neutral",
                    role="perturbation_solver",
                    model=model,
                    prompt=prompt_false_answer_neutral(statement=statement, false_answer=false_answer),
                    expects_json=False,
                    metadata=common,
                )
            )
            jobs.append(
                make_job(
                    index=len(jobs),
                    stage="false_answer_pressure",
                    role="perturbation_solver",
                    model=model,
                    prompt=prompt_false_answer_pressure(statement=statement, false_answer=false_answer),
                    expects_json=False,
                    metadata=common,
                )
            )
    return jobs


def build_error_detection_jobs(rows: list[dict[str, Any]], *, models: list[str]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        statement = statement_for(row)
        solution = solution_for(row)
        if not (statement and solution):
            continue
        for model in models:
            jobs.append(
                make_job(
                    index=len(jobs),
                    stage="error_detection",
                    role="judge",
                    model=model,
                    prompt=prompt_error_detection(statement=statement, solution=solution),
                    expects_json=True,
                    metadata={"record_id": record_id(row, row_index)},
                )
            )
    return jobs


def summarize_jobs(jobs: list[dict[str, Any]], *, skipped_rows: int = 0) -> dict[str, Any]:
    return {
        "jobs": len(jobs),
        "skipped_rows": skipped_rows,
        "by_stage": dict(sorted(Counter(str(job["stage"]) for job in jobs).items())),
        "by_role": dict(sorted(Counter(str(job["role"]) for job in jobs).items())),
        "by_model": dict(sorted(Counter(str(job["model"]) for job in jobs).items())),
    }


def parse_int_csv(value: str) -> list[int]:
    parsed = [int(item) for item in split_csv(value)]
    if not parsed:
        raise argparse.ArgumentTypeError("Expected at least one integer")
    return parsed


def build_jobs_from_args(args: argparse.Namespace) -> list[dict[str, Any]]:
    models = split_csv(args.models)
    if not models:
        raise ValueError("--models is required")
    validate_external_models(models, allow_student_lineage=args.allow_student_lineage)

    if args.stage == "seed":
        return build_seed_jobs(
            models=models,
            domains=split_csv(args.domains) or ["math"],
            difficulties=split_csv(args.difficulties) or ["medium"],
            target_steps=args.target_steps,
            count_per_combo=args.count_per_combo,
        )

    if not args.input_jsonl:
        raise ValueError(f"--input_jsonl is required for stage {args.stage!r}")
    rows = read_jsonl(args.input_jsonl)
    if args.stage == "ground_truth":
        return build_ground_truth_jobs(rows, models=models)
    if args.stage == "method_solve":
        return build_method_solve_jobs(rows, models=models, fallback_methods=split_csv(args.methods))
    if args.stage == "naturalness":
        return build_naturalness_jobs(rows, models=models)
    if args.stage == "depth":
        return build_depth_jobs(rows, models=models)
    if args.stage == "perturbation":
        return build_perturbation_jobs(rows, models=models)
    if args.stage == "error_detection":
        return build_error_detection_jobs(rows, models=models)
    raise ValueError(f"Unsupported stage {args.stage!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=("seed", "ground_truth", "method_solve", "naturalness", "depth", "perturbation", "error_detection"),
    )
    parser.add_argument("--models", required=True, help="Comma-separated external model ids.")
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--report_json")
    parser.add_argument("--input_jsonl")
    parser.add_argument("--domains", default="math")
    parser.add_argument("--difficulties", default="medium")
    parser.add_argument("--target_steps", type=parse_int_csv, default=[4])
    parser.add_argument("--count_per_combo", type=int, default=1)
    parser.add_argument("--methods", help="Optional comma-separated method names for method_solve.")
    parser.add_argument("--allow_student_lineage", action="store_true")
    args = parser.parse_args(argv)

    jobs = build_jobs_from_args(args)
    write_jsonl(args.output_jsonl, jobs)
    report = summarize_jobs(jobs)
    if args.report_json:
        out = Path(args.report_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"jobs={report['jobs']}")
    print(f"by_stage={report['by_stage']}")
    print(f"by_role={report['by_role']}")
    print(f"by_model={report['by_model']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

