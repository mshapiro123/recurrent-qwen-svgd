"""Assemble measured curriculum artifacts into typed curriculum records.

Inputs are verified candidates, correct-answer method solution candidates,
naturalness judgments, and depth measurements. Outputs are typed records safe to
pass through ``training/prepare_curriculum_jsonl.py`` before SFT.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from training.prepare_curriculum_jsonl import validate_curriculum_record


AUXILIARY_ROLE_PREFIXES = ("negative_", "verifier_")


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


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_label(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def index_by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        key = str(row.get("id") or row.get("record_id") or f"row-{index:06d}")
        indexed[key] = row
    return indexed


def target_loop_count(mode: str, min_steps: int, *, max_target_loops: int = 4) -> int:
    if mode == "direct":
        return 1
    return max(2, min(max_target_loops, 1 + (min_steps + 1) // 3))


def infer_mode(width: int, min_steps: int, *, deep_threshold: int) -> str:
    is_deep = min_steps >= deep_threshold
    if width >= 2 and is_deep:
        return "both"
    if width >= 2:
        return "wide"
    if is_deep:
        return "deep_narrow"
    return "direct"


def natural_solution_ids(
    judgments: list[dict[str, Any]],
    *,
    min_natural_agree: int,
    require_method_match: bool = True,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for judgment in judgments:
        grouped[str(judgment.get("solution_id") or "")].append(judgment)

    accepted: dict[str, dict[str, Any]] = {}
    for solution_id, rows in grouped.items():
        method = normalize_label(rows[0].get("method"))
        agreeing = []
        for row in rows:
            if row.get("natural") is not True:
                continue
            actually_uses = normalize_label(row.get("actually_uses"))
            if require_method_match and actually_uses and actually_uses != method:
                continue
            agreeing.append(row)
        agreeing_models = sorted({str(row.get("judge_model")) for row in agreeing})
        if len(agreeing_models) >= min_natural_agree:
            accepted[solution_id] = {
                "agreeing_models": agreeing_models,
                "judgments": agreeing,
            }
    return accepted


def depth_by_solution(measurements: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for measurement in measurements:
        grouped[str(measurement.get("solution_id") or "")].append(measurement)

    selected: dict[str, dict[str, Any]] = {}
    for solution_id, rows in grouped.items():
        valid = [
            row
            for row in rows
            if isinstance(row.get("count"), int) and not isinstance(row.get("count"), bool) and row["count"] > 0
        ]
        if not valid:
            continue
        best = min(valid, key=lambda row: (int(row["count"]), str(row.get("judge_model"))))
        selected[solution_id] = {
            "count": int(best["count"]),
            "steps": best.get("steps") if isinstance(best.get("steps"), list) else [],
            "judge_model": best.get("judge_model"),
            "measurements": valid,
        }
    return selected


def pair_key(left: str, right: str) -> tuple[str, str]:
    ordered = sorted((left, right))
    return ordered[0], ordered[1]


def distinct_solution_pairs(
    judgments: list[dict[str, Any]],
    *,
    min_distinct_agree: int,
) -> set[tuple[str, str]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for judgment in judgments:
        solution_a = str(judgment.get("solution_a_id") or "")
        solution_b = str(judgment.get("solution_b_id") or "")
        if not (solution_a and solution_b):
            continue
        grouped[pair_key(solution_a, solution_b)].append(judgment)

    accepted: set[tuple[str, str]] = set()
    for key, rows in grouped.items():
        true_models = {
            str(row.get("judge_model"))
            for row in rows
            if row.get("distinct") is True and str(row.get("judge_model"))
        }
        false_models = {
            str(row.get("judge_model"))
            for row in rows
            if row.get("distinct") is False and str(row.get("judge_model"))
        }
        if len(true_models) >= min_distinct_agree and len(true_models) > len(false_models):
            accepted.add(key)
    return accepted


def solution_id(solution: dict[str, Any]) -> str:
    return str(solution.get("id") or "")


def prune_to_pairwise_distinct_methods(
    best_by_method: dict[str, tuple[dict[str, Any], dict[str, Any], dict[str, Any]]],
    accepted_pairs: set[tuple[str, str]],
) -> dict[str, tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]:
    selected: dict[str, tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = {}
    for method, triple in sorted(best_by_method.items(), key=lambda item: (int(item[1][2]["count"]), item[0])):
        current_id = solution_id(triple[0])
        if not current_id:
            continue
        is_distinct_from_selected = all(
            pair_key(current_id, solution_id(selected_triple[0])) in accepted_pairs
            for selected_triple in selected.values()
        )
        if is_distinct_from_selected:
            selected[method] = triple
    return selected


def build_trace(solution: dict[str, Any], *, role: str, steps: int, natural_info: dict[str, Any]) -> dict[str, Any]:
    trace = {
        "role": role,
        "method": solution.get("method"),
        "correct": True,
        "natural": True,
        "steps": steps,
        "source_model": solution.get("source_model"),
        "logical_source_model": solution.get("logical_source_model"),
        "source_job_id": solution.get("source_job_id"),
        "source_response_id": solution.get("source_response_id"),
        "naturalness_judges": natural_info.get("agreeing_models", []),
        "text": str(solution.get("text") or solution.get("solution") or ""),
    }
    answer_match = solution.get("answer_match")
    if isinstance(answer_match, dict):
        trace["answer_match"] = answer_match
    return trace


def build_auxiliary_trace(row: dict[str, Any]) -> dict[str, Any]:
    trace: dict[str, Any] = {
        "role": row.get("role"),
        "correct": row.get("correct"),
        "source_model": row.get("source_model"),
        "logical_source_model": row.get("logical_source_model"),
        "source_job_id": row.get("source_job_id"),
        "source_response_id": row.get("source_response_id"),
        "text": str(row.get("text") or row.get("solution") or row.get("explanation") or ""),
    }
    for key in (
        "detected",
        "error_type",
        "injected",
        "false_answer",
        "first_error_step",
        "verdict",
        "explanation",
        "source_trace_id",
    ):
        if key in row:
            trace[key] = row[key]
    return trace


def is_safe_auxiliary_trace(row: dict[str, Any]) -> bool:
    role = str(row.get("role") or "").strip()
    return role.startswith(AUXILIARY_ROLE_PREFIXES)


def assemble_curriculum_records(
    verified_candidates: list[dict[str, Any]],
    solution_candidates: list[dict[str, Any]],
    naturalness_judgments: list[dict[str, Any]],
    depth_measurements: list[dict[str, Any]],
    distinctness_judgments: list[dict[str, Any]] | None = None,
    auxiliary_traces: list[dict[str, Any]] | None = None,
    *,
    min_natural_agree: int = 1,
    min_distinct_agree: int = 1,
    deep_threshold: int = 5,
    require_decontaminated: bool = True,
    require_method_match: bool = True,
    max_target_loops: int = 4,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidate_by_id = index_by_id(verified_candidates)
    natural = natural_solution_ids(
        naturalness_judgments,
        min_natural_agree=min_natural_agree,
        require_method_match=require_method_match,
    )
    depth = depth_by_solution(depth_measurements)
    distinct_pairs = (
        distinct_solution_pairs(distinctness_judgments, min_distinct_agree=min_distinct_agree)
        if distinctness_judgments is not None
        else None
    )
    solutions_by_record: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for solution in solution_candidates:
        solutions_by_record[str(solution.get("record_id") or "")].append(solution)
    auxiliary_by_record: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trace in auxiliary_traces or []:
        auxiliary_by_record[str(trace.get("record_id") or "")].append(trace)

    records: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    unsafe_auxiliary: list[dict[str, Any]] = []
    validation_issues: list[str] = []

    for record_id, candidate in sorted(candidate_by_id.items()):
        if require_decontaminated and candidate.get("decontaminated") is not True:
            rejected.append({"id": record_id, "reason": "not_decontaminated"})
            continue

        accepted_solutions: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
        for solution in solutions_by_record.get(record_id, []):
            solution_id = str(solution.get("id") or "")
            natural_info = natural.get(solution_id)
            depth_info = depth.get(solution_id)
            if natural_info and depth_info:
                accepted_solutions.append((solution, natural_info, depth_info))

        if not accepted_solutions:
            rejected.append({"id": record_id, "reason": "no_natural_depth_measured_solution"})
            continue

        per_method: dict[str, int] = {}
        best_by_method: dict[str, tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = {}
        for solution, natural_info, depth_info in accepted_solutions:
            method = str(solution.get("method") or "").strip()
            if not method:
                continue
            count = int(depth_info["count"])
            if method not in per_method or count < per_method[method]:
                per_method[method] = count
                best_by_method[method] = (solution, natural_info, depth_info)

        if distinct_pairs is not None:
            best_by_method = prune_to_pairwise_distinct_methods(best_by_method, distinct_pairs)
            per_method = {
                method: int(depth_info["count"])
                for method, (_solution, _natural_info, depth_info) in best_by_method.items()
            }

        if not per_method:
            rejected.append({"id": record_id, "reason": "no_method_depth"})
            continue

        methods = sorted(per_method)
        min_steps = min(per_method.values())
        mode = infer_mode(len(methods), min_steps, deep_threshold=deep_threshold)
        role = "positive_wide" if mode in {"wide", "both"} else ("positive_depth" if mode == "deep_narrow" else "positive_direct")
        traces = [
            build_trace(solution, role=role, steps=int(depth_info["count"]), natural_info=natural_info)
            for method, (solution, natural_info, depth_info) in sorted(best_by_method.items())
        ]
        for trace in auxiliary_by_record.get(record_id, []):
            if not is_safe_auxiliary_trace(trace):
                unsafe_auxiliary.append(
                    {
                        "record_id": record_id,
                        "id": trace.get("id"),
                        "role": trace.get("role"),
                        "reason": "auxiliary traces must be negative_ or verifier_ roles",
                    }
                )
                continue
            traces.append(build_auxiliary_trace(trace))
        answer = candidate.get("answer") if isinstance(candidate.get("answer"), dict) else {}
        record = {
            "id": record_id,
            "domain": candidate.get("domain") or "unknown",
            "statement": candidate.get("statement"),
            "answer": answer,
            "difficulty": candidate.get("difficulty")
            if isinstance(candidate.get("difficulty"), dict)
            else {
                "pass_rate": candidate.get("difficulty_pass_rate"),
                "reference_model": "unmeasured",
            },
            "width_signature": {"methods": methods, "width": len(methods)},
            "depth": {"per_method": per_method, "min_steps": min_steps},
            "mode": mode,
            "target_loop_count": target_loop_count(mode, min_steps, max_target_loops=max_target_loops),
            "decontaminated": True,
            "source_dataset": "external_strong_model_curriculum",
            "traces": traces,
        }
        issues = validate_curriculum_record(record)
        if issues:
            validation_issues.extend(f"{record_id}: {issue}" for issue in issues)
            rejected.append({"id": record_id, "reason": "validation_failed", "issues": issues})
            continue
        records.append(record)

    report = {
        "mode": "assembled_curriculum_records",
        "verified_candidates": len(verified_candidates),
        "solution_candidates": len(solution_candidates),
        "naturalness_judgments": len(naturalness_judgments),
        "depth_measurements": len(depth_measurements),
        "distinctness_judgments": 0 if distinctness_judgments is None else len(distinctness_judgments),
        "distinctness_required": distinctness_judgments is not None,
        "min_natural_agree": min_natural_agree,
        "min_distinct_agree": min_distinct_agree,
        "auxiliary_traces": 0 if auxiliary_traces is None else len(auxiliary_traces),
        "unsafe_auxiliary_traces": len(unsafe_auxiliary),
        "unsafe_auxiliary_trace_rows": unsafe_auxiliary,
        "records": len(records),
        "rejected": len(rejected),
        "rejected_records": rejected,
        "validation_issues": validation_issues,
        "mode_counts": dict(sorted(Counter(str(record["mode"]) for record in records).items())),
        "target_loop_counts": dict(
            sorted(
                Counter(str(record.get("target_loop_count")) for record in records).items(),
                key=lambda item: int(item[0]) if str(item[0]).isdigit() else 999,
            )
        ),
    }
    return records, report


def write_report(path: str | Path | None, report: dict[str, Any]) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verified_candidates_jsonl", required=True)
    parser.add_argument("--solution_candidates_jsonl", required=True)
    parser.add_argument("--naturalness_jsonl", required=True)
    parser.add_argument("--depth_jsonl", required=True)
    parser.add_argument("--distinctness_jsonl")
    parser.add_argument("--auxiliary_traces_jsonl", action="append")
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--report_json")
    parser.add_argument("--min_natural_agree", type=int, default=2)
    parser.add_argument("--min_distinct_agree", type=int, default=2)
    parser.add_argument("--deep_threshold", type=int, default=5)
    parser.add_argument("--allow_not_decontaminated", action="store_true")
    parser.add_argument("--allow_method_mismatch", action="store_true")
    parser.add_argument("--max_target_loops", type=int, default=4)
    args = parser.parse_args(argv)

    records, report = assemble_curriculum_records(
        read_jsonl(args.verified_candidates_jsonl),
        read_jsonl(args.solution_candidates_jsonl),
        read_jsonl(args.naturalness_jsonl),
        read_jsonl(args.depth_jsonl),
        read_jsonl(args.distinctness_jsonl) if args.distinctness_jsonl else None,
        [row for path in (args.auxiliary_traces_jsonl or []) for row in read_jsonl(path)],
        min_natural_agree=args.min_natural_agree,
        min_distinct_agree=args.min_distinct_agree,
        deep_threshold=args.deep_threshold,
        require_decontaminated=not args.allow_not_decontaminated,
        require_method_match=not args.allow_method_mismatch,
        max_target_loops=args.max_target_loops,
    )
    write_jsonl(args.output_jsonl, records)
    write_report(args.report_json, report)
    print(f"records={report['records']}")
    print(f"rejected={report['rejected']}")
    print(f"mode_counts={report['mode_counts']}")
    print(f"validation_issues={len(report['validation_issues'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
